import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AddEditTechComponent } from './add-edit-tech.component';

describe('AddEditTechComponent', () => {
  let component: AddEditTechComponent;
  let fixture: ComponentFixture<AddEditTechComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AddEditTechComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AddEditTechComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
